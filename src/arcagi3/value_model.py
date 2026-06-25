from __future__ import annotations

import numpy as np


class ValueModel:
    def __init__(self, input_dim: int, lr: float = 0.1):
        self.w = np.zeros(input_dim, dtype=float)
        self.b = 0.0
        self.lr = float(lr)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def score(self, features) -> float:
        x = np.asarray(features, dtype=float)
        return float(self._sigmoid(np.dot(self.w, x) + self.b))

    def fit(self, rows, epochs: int = 20):
        for _ in range(int(epochs)):
            for features, label in rows:
                x = np.asarray(features, dtype=float)
                y = float(label)
                pred = self.score(x)
                err = pred - y
                self.w -= self.lr * err * x
                self.b -= self.lr * err
